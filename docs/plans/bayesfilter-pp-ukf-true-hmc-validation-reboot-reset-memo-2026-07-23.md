# PP-UKF true-HMC validation reboot reset memo

Date: 2026-07-23 (Asia/Hong_Kong)

Repository commit observed at terminal run: `4281adf3c6067b706d83841bfc7a8fba022a65dd`

Campaign plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`

## Terminal state

Attempt 09 finished normally in the detached session
`pp_ukf_hmc_24h_20260722_restart`. The tmux session and Python process are no
longer running. The terminal artifacts are:

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/public_result.json
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/run_manifest.json
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json
```

Attempt 09 recorded exit code `0`, `10/10` candidate rows, no hard vetoes, and
aggregate charged time:

| Quantity | Seconds |
|---|---:|
| Carry-in charge from attempts 01--08 | 17,424.711535 |
| Attempt 09 wall time | 24,978.793005 |
| Aggregate charge | 42,403.504540 |
| Authorized cap | 86,400.000000 |
| Remaining budget | 43,996.495460 |

The attempt-08 resume checkpoint remains unchanged:

```text
docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-08/progress.json
SHA-256: 8590b64b48581d3bb13f8a8c02aa2dee323d6d842a87f83909a8063d3a5c391d
```

The frozen target and transport identities also remain binding:

```text
target_signature: d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5
transport_sha256: b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221
```

## Attempt-09 evidence

| `L` | Terminal decision | Retained per chain | Final max R-hat | Final min bulk ESS | Final min tail ESS |
|---:|---|---:|---:|---:|---:|
| 5 | admitted | 1,000 | 1.0059 | 3,306 | 1,813 |
| 9 | rejected at retained screen | 3,000 | 1.0314 | 6,885 | 1,061 |
| 12 | rejected at retained screen | 3,000 | 1.0168 | 11,162 | 2,503 |
| 13 | admitted | 1,000 | 1.0050 | 1,923 | 1,305 |
| 14 | admitted | 2,000 | 1.0058 | 1,359 | 2,434 |
| 17 | rejected at retained screen | 3,000 | 1.0199 | 4,901 | 1,689 |
| 18 | admitted | 1,000 | 1.0087 | 1,708 | 916 |
| 19 | admitted | 2,000 | 1.0046 | 1,243 | 1,935 |
| 24 | admitted | 1,500 | 1.0069 | 1,463 | 1,108 |
| 25 | admitted | 2,000 | 1.0073 | 2,665 | 1,390 |

The seven admissions are viable under the observed early stopping points. No
statistical ranking was performed. Acceptance, runtime, and finite extreme
log-acceptance counts remain explanatory only. Native TFP divergence is not
exposed and is recorded as unavailable, never as zero.

## Protocol defect requiring repair

The reviewed plan and repository NeuTra policy specify a retained maximum of
10,000 transitions per chain. The attempt-09 driver used:

```python
MAX_RESULTS = 3_000
warmup_max_results=MAX_RESULTS
retained_max_results=MAX_RESULTS
```

Therefore `L=9,12,17` are not final scientific rejections under the declared
protocol. They failed the retained screen at 3,000 draws, but the run did not
allow the planned 10,000-draw continuation. This is a harness/protocol defect,
not evidence against those candidates or against PP-UKF HMC.

The driver also removes any resumed candidate by candidate ID regardless of its
prior `passed` value. A repair continuation must not silently treat a failed
short-cap row as complete. It must either use an explicit candidate index list
or require a new progress schema that distinguishes `admitted`, `rejected_at_cap`,
and `incomplete` rows.

## Reboot contract

Do not overwrite attempts 08 or 09. Do not launch a reboot until the driver and
focused tests establish all of the following:

1. `retained_max_results=10_000` and warm-up maximum remains consistent with the
   reviewed policy.
2. The continuation selects exactly `L=9,12,17`; completed admissions are not
   rerun.
3. The continuation writes a fresh output root and preserves all prior rows by
   reference, without mutating attempt-08 or attempt-09.
4. A candidate failing at 3,000 is allowed to continue to 10,000, while a
   candidate passing earlier may stop at the declared minimum.
5. The result and manifest record the repaired policy, candidate indices,
   carry-in progress hashes, aggregate budget, GPU memory policy, target and
   transport identities, and the native-divergence limitation.

The next fresh root is `attempt-10`. A command skeleton, to be used only after
the repair tests pass, is:

```bash
tmux new-session -d -s pp_ukf_hmc_24h_20260723_reboot \
  "cd /home/chakwong/BayesFilter && \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
    docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py \
    --output-root docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10 \
    --candidate-index 1 --candidate-index 2 --candidate-index 5 \
    --resume-progress docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json \
    --prior-elapsed-seconds 42403.504540 > \
    docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10-launch.log 2>&1"
```

The candidate indices correspond to the frozen manifest order
`(5,9,12,13,14,17,18,19,24,25)`. If the repaired driver changes the progress
schema or CLI, update this command in the execution note before launch. The
remaining budget is about `43,996.495460` seconds; stop before the aggregate
charge reaches `86,400` seconds.

## Decision and inference status

| Item | Status | Reboot implication |
|---|---|---|
| Engineering run completion | complete | preserve attempt 09; create attempt 10 only after repair |
| Hard-veto screen | no hard vetoes in attempt 09 | no direction veto |
| Declared retained protocol | not satisfied for `L=9,12,17` | continue those rows to the 10,000 cap |
| Viable candidate set | `L=5,13,14,18,19,24,25` | retain as unranked viable evidence |
| Statistically supported ranking | none | do not rank candidates |
| Posterior correctness/default readiness | not established | requires separate downstream evidence |

Strongest alternative explanation: the three rejected rows may converge with a
longer retained ladder, so their current rejection is censoring by an
implementation cap rather than a candidate property. The reboot repairs that
specific ambiguity and changes no target, data, frozen controls, or scientific
question.
