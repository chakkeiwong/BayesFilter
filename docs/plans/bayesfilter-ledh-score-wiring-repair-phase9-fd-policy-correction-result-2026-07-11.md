# Phase 9 FD Policy Correction Result

Date: 2026-07-11

Status: `CORRECTION_COMPLETE_OWNER_DIRECTED_POLICY_RECLASSIFIED`

## Outcome

The former Phase 9 hard-veto decisions based on the inherited `0.005`
absolute-or-relative gate and the intervening `2%` RSS/RMS correction are
superseded. The owner clarified that this is an FD-only check over individual
parameter directions with a `5% * sqrt(p)` tolerance. The original trusted
GPU/XLA score and FD values remain valid raw measurements and were not modified.

The corrected rule classifies 9 of 11 stored comparisons as passing and
2 as failing. Predator-prey fails at Gate B and
generalized-SV passes Gate B but fails at Gate C. Fixed-SIR, Actual-SV, and
KSC-SV have no stored FD failure under this rule. This is an offline
reclassification of stored values, not a new GPU run.

## Corrected Policy

For each direction, preserve the historical coordinate definition
`r_j = |score_j - FD_j| / max(|score_j|, |FD_j|, 1e-12)`. For `p`
parameter directions, a comparison passes exactly when
`max_j(r_j) <= 0.05 * sqrt(p)`. Directions are not combined with RSS, RMS,
or an average, and there is no absolute-error escape hatch.

The `5%` choice mirrors the conventional 95% confidence/significance
threshold. The FD calculation is not itself a confidence interval: no
sampling distribution, standard error, coverage calculation, or repeated-run
calibration is computed.

## Stored FD Stops

| Row | Rung | p | Maximum direction error | `0.05*sqrt(p)` | Maximum-error parameter | FD decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| predator-prey | T=1,N=2 | 6 | 1 | 0.122474487139 | a | FAIL |
| generalized-sv | T=4,N=10000 | 3 | 0.442753962161 | 0.0866025403784 | log_tau | FAIL |

## Historical Terminal Shards

| Row | Historical rung | p | Maximum direction error | Threshold | FD decision |
| --- | --- | ---: | ---: | ---: | --- |
| predator-prey | T=1,N=2 | 6 | 1 | 0.122474487139 | fail |
| fixed-sir | T=20,N=10000 | 3 | 0.0566700085587 | 0.0866025403784 | pass |
| actual-sv | T=4,N=10000 | 2 | 0.0602924688125 | 0.0707106781187 | pass |
| generalized-sv | T=4,N=10000 | 3 | 0.442753962161 | 0.0866025403784 | fail |
| ksc-sv | T=4,N=10000 | 2 | 0.0369351492982 | 0.0707106781187 | pass |

## All Reclassified Rungs

| ID | Legacy decision (superseded basis) | Maximum direction error | Threshold | FD decision |
| --- | --- | ---: | ---: | --- |
| gate-b-fixed-sir-t1-n4-seed81120 | pass | 0.0170331796823 | 0.0866025403784 | pass |
| gate-b-predator-prey-t1-n2-seed81120 | fail | 1 | 0.122474487139 | fail |
| gate-b-actual-sv-t1-n4-seed81120 | pass | 0.0191197815534 | 0.0707106781187 | pass |
| gate-b-generalized-sv-t1-n4-seed81120 | pass | 0.0416117625727 | 0.0866025403784 | pass |
| gate-b-ksc-sv-t1-n4-seed81120 | pass | 0.00310435126582 | 0.0707106781187 | pass |
| gate-c-fixed-sir-t1-n10000-seed81120 | pass | 0.0171573614557 | 0.0866025403784 | pass |
| gate-c-fixed-sir-t5-n10000-seed81120 | pass | 0.00206817823589 | 0.0866025403784 | pass |
| gate-c-fixed-sir-t20-n10000-seed81120 | fail | 0.0566700085587 | 0.0866025403784 | pass |
| gate-c-actual-sv-t4-n10000-seed81120 | fail | 0.0602924688125 | 0.0707106781187 | pass |
| gate-c-generalized-sv-t4-n10000-seed81120 | fail | 0.442753962161 | 0.0866025403784 | fail |
| gate-c-ksc-sv-t4-n10000-seed81120 | fail | 0.0369351492982 | 0.0707106781187 | pass |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Supersede both the old `0.005` decisions and the wrong `2%` RSS/RMS correction. | Nine stored comparisons pass the FD-only rule; predator-prey Gate B and generalized-SV Gate C fail it. Actual-SV passes with `0.0602925 <= 0.0707107`. | Source hashes, FD-to-score bindings, parameter order, finiteness, and legacy stored-field consistency passed. | Float32 FD resolution versus compact-score math remains unisolated for the two failures; comparisons use one seed and one FD step. | Treat the three newly passing historical terminal rows as having this FD veto removed; separately plan any continuation required by the original ladder. Diagnose the two failures only under a reviewed derivative-resolution plan. | No general score correctness, HMC readiness, posterior correctness, default readiness, full admission, causal attribution, calibrated confidence interval, or superiority. |

## Inference Status

| Item | Status |
| --- | --- |
| FD-only veto screen | Stored FD failures are supported only for predator-prey Gate B and generalized-SV Gate C. Fixed-SIR, Actual-SV, and KSC-SV have no stored failure under the clarified rule. |
| Statistically supported ranking | None. This deterministic reclassification provides no uncertainty analysis or candidate ranking. |
| Descriptive-only differences | Per-coordinate errors and margins, FD steps, runtime, and memory differences remain descriptive. |
| Default-readiness | Not established. Passing this FD diagnostic does not establish any broader readiness claim. |
| Next evidence needed | Original downstream ladder requirements remain separate. The two FD failures need a reviewed precision/step diagnostic if pursued. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Reclassifier completed and verified every source hash and cross-shard binding. |
| Numerical validity | Stored comparisons were re-evaluated with the maximum individual-direction formula; nine pass and two fail. |
| Scientific interpretation | FD diagnostic only. General score validity, HMC behavior, and posterior validity were not evaluated. |

## Run Manifest

- JSON artifact: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`
- git_commit: `d269f5bbd8531b878d4f25897a357fbc8f172488`
- command: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python docs/benchmarks/reclassify_ledh_phase9_fd_policy.py --manifest docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json --markdown-output docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`
- working_directory: `/home/chakwong/BayesFilter`
- python_executable: `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`
- python_version: `3.11.14 (main, Oct 21 2025, 18:31:21) [GCC 11.2.0]`
- host: `DESKTOP-RF1Q5IJ`
- platform: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`
- conda_prefix: `/home/chakwong/anaconda3/envs/tf-gpu`
- input_manifest: `docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json`
- input_manifest_sha256: `83da76f12f16d915c2f9f6722ce4e65fc91c81fddd90acb696a12358162eb044`
- output_json: `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`
- output_markdown: `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`
- reclassifier_path: `docs/benchmarks/reclassify_ledh_phase9_fd_policy.py`
- reclassifier_sha256: `ffe0277122e8ba15e99992e6ae27723609ae3fbe1087d816df956e8ce93fa897`
- policy_module_path: `bayesfilter/ledh_fd_policy.py`
- policy_module_sha256: `e1999cbe08312048abd2eab30c15dc33adaf520b5d745f47e139aac3eac79da2`
- plan_path: `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md`
- plan_sha256: `935f27eea038813442be2cde7cd9ceece9a2827ed7dc6852a1c999476ad9d12a`
- historical_result_path: `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md`
- execution_target: `cpu_only_offline_reclassification`
- cuda_visible_devices: `-1`
- gpu_execution_performed: `False`
- random_seeds: `preserved from source shards; no new randomness`
- wall_time_seconds: `0.01887111086398363`
- data_version: `SHA-256-bound immutable Phase 9 score/FD JSON shards`
- git_status_short: full dirty-worktree disclosure is preserved in the JSON artifact

## Source Bindings

| ID | Score JSON SHA-256 | FD JSON SHA-256 |
| --- | --- | --- |
| gate-b-fixed-sir-t1-n4-seed81120 | `0f8a71afdce3571e952c061f036efd66923c08c42392b08999d1fb72c7631f18` | `1e1c3ceeb9e793298903b6853db463abfca13d41e73fb0126037807cdfe759be` |
| gate-b-predator-prey-t1-n2-seed81120 | `82eb75a8710a6c4219419b5f9c14f670e371554a3c7943a2a3fb5e03f1c28f5c` | `738c59f9967ec86dfc09be7bfb315e4cc9fdfc04a22cec95292527405f1b3127` |
| gate-b-actual-sv-t1-n4-seed81120 | `ad697c4bd06749cab022281b128b772e61286c0c8a506de623e49b0f03ea6e55` | `c5cda3c38e05564aa011168353b7f519bb1ce676b02127401f2fa618c41c6a92` |
| gate-b-generalized-sv-t1-n4-seed81120 | `db0891693d55faa77e0dac689b26bf8d8394d802dd724847ff5ba3849a27a7d8` | `4fa4f8280b5d0f2d4ced2f8c048a79c0f358d16ab41874fd22d0020ec5a327ce` |
| gate-b-ksc-sv-t1-n4-seed81120 | `832d183455de798c6794594a87e4c6b5630dc36201b7000e399342b5077cdbf6` | `2fc17704717383352a06362a951b7d3b8b38c1c06a9aa7d2cf712d0e2c04b641` |
| gate-c-fixed-sir-t1-n10000-seed81120 | `ca991fa7c4dad820d715fcecedc3f591c5d52a144308cc68bdf2d5287240e794` | `606c194d7dc2d7264bd41f1f7a0b7ebcf08219fb74ff429ca97492280fbc1c4a` |
| gate-c-fixed-sir-t5-n10000-seed81120 | `2a19ef266749e876f741a0955556d92d56cf15178c041585ac9671fa3aa878f1` | `18cc1c4fc18bb1800109cf54440fa33cadab3f0367f34349e4d50afb4b09b299` |
| gate-c-fixed-sir-t20-n10000-seed81120 | `7acf4612b4082533cfa076635f1788015ffae43da94f15eb4e818e57c2036773` | `00944bcb7f756f914b56f920b62709e9c4d9a950b5dffcf8589ac83fd68f0036` |
| gate-c-actual-sv-t4-n10000-seed81120 | `6320f04eab3f03157e3c1789de5b1927cefb33c9752e2fb0a7cfe787797f86b7` | `9547b853db09e2974f2dfa2adf8d5d3d19b274b5dfb74dab8358895e8b03bdaf` |
| gate-c-generalized-sv-t4-n10000-seed81120 | `3fb140284b74a02efb8fe57562f0f33ee75a1012bd1dd6cdc554be71c59e71d6` | `edc896ef4a41772e29487257b3c6e01c8543b780aacf930fa607dd43479b8b08` |
| gate-c-ksc-sv-t4-n10000-seed81120 | `232c28ae76c945efc843f296e412f58ef30d3db38e28e958d47be633c9311dae` | `288f997acb7dcc0440a5bcd653e34ca626884fc1421e35e2c3fd25048bde366d` |

## Post-Run Red Team

- Strongest alternative explanation: production-float32 central FD at
  the historical step may be resolution-limited. This remains explanatory
  because no precision or step ladder was run.
- Result that would overturn either stored FD failure: a reviewed,
  predeclared same-scalar derivative check of the unchanged target that
  passes this corrected policy with source-bound evidence.
- Weakest evidence: each comparison is one seed at one FD step; there is
  no calibrated confidence interval despite the conventional 5% motivation.
- The correction does not authorize a GPU rerun, Gate D, aggregation,
  HMC execution, or Phase 10.
