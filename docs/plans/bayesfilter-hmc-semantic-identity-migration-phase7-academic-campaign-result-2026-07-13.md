# Phase 7 Academic Serious-HMC Campaign Result

Date: 2026-07-13

Status: `TERMINAL_DIAGNOSTIC_CAP_FAILURE_NO_RETRY`

## Outcome

The fixed typed-identity deterministic LGSSM HMC campaign executed successfully
as an engineering workload but did not pass its scientific promotion screen.
Attempt 1 reached the declared burn-in cap of `16000` transitions per chain.
The final diagnostic window was finite and passed both ESS thresholds, but
eight of 18 parameters exceeded R-hat `1.01`. Retained sampling did not begin.

The terminal classification is `diagnostic_cap_failure`, not infrastructure
failure. The current campaign is therefore terminal, no retry is permitted,
and Phase 8 posterior-recovery/runtime work and NeuTra remain blocked.

## Research Intent Ledger

| Field | Recorded value |
| --- | --- |
| Main question | Can the fixed typed transition complete serious burn-in and retained sampling and pass every all-parameter convergence gate? |
| Candidate | Transition identity `sha256:10d9a9d2d71562d0c278b5bbc0ba0bb3eed3fc2ae77510a6d09e5c16a6f16d6a` under serious execution identity `sha256:ceefd154f97510f2b432c45287a0f309792a3def3855dec3ffd2061f2b4587e4` |
| Expected failure mode | One or more parameters fail R-hat, bulk ESS, or tail ESS at a declared burn-in or retained cap |
| Promotion criterion | Every parameter has R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`, with all engineering/numerical vetoes clear |
| Promotion veto that fired | Eight parameters exceeded R-hat `1.01` at the burn-in cap |
| Continuation veto | The declared burn-in cap was reached without passing; retained sampling was correctly not started |
| Repair trigger | None inside this fixed campaign; the terminal result was not infrastructure failure |
| Must not be concluded | No target invalidity, posterior recovery, calibrated uncertainty, sampler ranking, production/default/GPU readiness, Phase 8 success, NeuTra readiness, or broad scientific invalidity |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Command | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 /home/chakwong/BayesFilter/scripts/run_hmc_phase7_academic_campaign.py` |
| Environment | conda `tf-gpu`; Python `3.11.14` |
| Device | Deliberate CPU-only run with `CUDA_VISIBLE_DEVICES=-1` |
| TensorFlow route | float64 Host XLA, `jit_compile=True`, `use_xla=True`; no non-JIT runtime |
| Topology | Two persistent workers, two chains per worker, four chains total |
| Root seed | `(20260711, 701)` |
| Thread environment | intra-op `8`; inter-op, OMP, OpenBLAS, and MKL each `1` |
| Config | `docs/benchmarks/configs/multidim_lgssm_phase7_typed_identity_baseline_2026_07_11.json` |
| Config file SHA-256 | `9270ec429a4b49e19f5ac6492e146bb1010e07c4ea0aa17600294e6c41db7ca8` |
| Config semantic hash | `sha256:79bcaa2b5977cadeb14607d6256e2eda31efb63d9d9a69d3008603fe14e3a450` |
| Data version | fixture file SHA-256 `49f308e445bd621e347ab4b2e364066327ec10d491fc248955867eea634f6913` |
| Attempt | `1` of at most `3`; terminal after non-infrastructure result |
| Controller elapsed | `535.8535413441714` seconds |
| Charged attempt/cumulative wall time | `541.4647207008675` seconds |
| Remaining nominal campaign budget | `28258.535279299133` seconds, unusable because the campaign is terminal |
| Budget overrun | `0.0` seconds |
| Plan | `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase7-academic-campaign-subplan-2026-07-13.md` |
| Result | This file |

The command was observed as a live foreground process in the current managed
Codex session. When the supervisor detected that attempt 1 already held the
campaign lock, it did not launch a duplicate. The supervisor then polled only
structured progress and process state until the terminal result appeared.

## Terminal Evidence

Run directory:
`docs/benchmarks/artifacts/multidim_lgssm_serious_hmc_tuning_2026_07_09/phase7_academic_campaign/attempt-001-20260712T214454Z`

| Artifact | Embedded hash | File SHA-256 |
| --- | --- | --- |
| Run manifest | `sha256:042496815298c958a540285c6bc22e3785a4f8173d272c78accf5aba71586da2` | `8f2682395fd9a70ec0150a475bbf7c4d1ec4a33a48a09550b6b62ef6060bd4d9` |
| Terminal result | `sha256:0724851756606956d2bf9d79fa62597fcef22a0c3c0737548d3383650306e076` | `2bb7f1f25810ba47c32a91841b69fac2ac0bc576e73ebc336008bc394bbabc4d` |
| Attempt summary | `sha256:6ed6256d8171fb6d4c1decad56632ee5113f4214def66ef4af09e059fd89838b` | `513c2838cb377af89f404ecc5be8360d8d49500179a2ae6e37a2c6771ac58daf` |
| Checksum manifest | `sha256:41f6682abc28edd8c3b5650db19b4a6ee906bf2cd40a34a5fcedb303a6cc0b0b` | `14c28d8ae791e0fdb5021baa20b706f3ac5abdcb6c4c53420170f4e5e9349c50` |
| Progress | N/A | `c04773eb8517575c57a74b7d0d8642f572c44759d34bd07d9832f22b73206584` |
| Log | N/A | `0732257e6a8b7400a685aac9bf73cebc4c70d80612b7c72384e1c7aa0fcca20b` |

`load_attempt_history` and `verify_checksum_manifest` both passed after the
run. No Phase 7 process remained. The private directory is empty by design
because burn-in failed before retained sampling; no retained-sample artifact
is claimed.

## Diagnostic Result

Final diagnostics used the last `1000` burn-in draws per chain, four chains,
and eight split chains with `500` draws each.

| Aggregate | Threshold | Observed | Status |
| --- | ---: | ---: | --- |
| Maximum R-hat | `<=1.01` | `1.043456525609825` | Fail |
| Minimum bulk ESS | `>=1000` | `1243.2342193161846` | Pass |
| Minimum tail ESS | `>=400` | `511.5036456092887` | Pass |
| Input/diagnostic finiteness | Required | Both true | Pass |
| Hard vetoes | None allowed | None | Pass |

The eight failed rows all failed R-hat only:

| Parameter | R-hat | Bulk ESS | Tail ESS |
| --- | ---: | ---: | ---: |
| `a22_raw` | `1.0272206846112695` | `4079.989724560392` | `1119.9663766654055` |
| `a33_raw` | `1.0105461682145447` | `3083.870666775378` | `825.7094449429574` |
| `a31_raw` | `1.0145997343916502` | `3476.550289723601` | `925.0004184478039` |
| `a32_raw` | `1.018760134460286` | `5907.889855675948` | `1105.4525338629105` |
| `a41_raw` | `1.043456525609825` | `6029.282646188154` | `734.3940654595516` |
| `a42_raw` | `1.0212535195271508` | `5469.053925237009` | `741.5660277698803` |
| `log_q1` | `1.013623145076172` | `2390.7937607615763` | `511.5036456092887` |
| `log_q3` | `1.0140839448728105` | `1675.002261611878` | `714.2113093794918` |

Intermediate burn-in metrics are explanatory only. R-hat did not pass at any
scheduled check from `2000` through `16000`, so the controller correctly
stopped at the cap.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Block fixed Phase 7 campaign | Failed: eight of 18 R-hat rows exceeded `1.01` at burn-in cap | Engineering/numerical validity vetoes clear; declared convergence cap fired | Whether failure is due to fixed transition tuning, initialization, last-window diagnostic behavior, or target geometry is not resolved | Close this runbook at the Phase 7 blocker; use a new reviewed research/repair plan if further diagnosis is desired | Target invalidity, HMC-direction rejection, posterior error, sampler inferiority, or production readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No crash, nonfinite value, reported hard veto, source/identity drift, XLA fallback, or corrupted terminal artifact |
| Statistically supported ranking | None; no method comparison or ranking was run |
| Descriptive-only differences | The per-check and per-parameter R-hat/ESS values describe this fixed run only |
| Default readiness | Not established; the candidate failed its fixed all-parameter convergence screen |
| Next evidence needed | A separate diagnostic plan that distinguishes transition tuning, initialization, diagnostic-window design, and target geometry before authorizing another serious campaign |

## Three Evidence Ledgers

| Ledger | Verdict |
| --- | --- |
| Engineering correctness | Passed for this run: typed identities, fixed config, CPU/XLA topology, terminal schemas, checksums, process teardown, and no-runtime regression gates verified |
| Sampler validity | Failed the declared burn-in convergence screen; retained-sample validity was not evaluated because retained sampling did not start |
| Scientific interpretation | The fixed candidate is rejected under this screen. The broader HMC direction and target are not rejected by this evidence |

## Verification And Review

- Focused academic wrapper/controller gate: `59 passed` during repair.
- Final focused academic gate: `31 passed`, two known TFP deprecation warnings.
- Complete identity/integration/certificate/adoption/smoke/serious/controller/
  academic no-runtime gate: `319 passed`, two known TFP deprecation warnings,
  in `416.49 s`.
- `py_compile` and `git diff --check`: passed.
- Historical authority files retained their fixed SHA-256 values.
- Independent Codex convergence review closed all four pre-launch findings and
  returned `VERDICT: AGREE`; no finding required stopping attempt 1.

## Negative-Result Classification

- Implementation failure: not supported.
- Tuning/candidate failure: supported for the fixed transition/execution
  candidate under the declared burn-in screen.
- Diagnostic failure: the diagnostic computation was finite and internally
  consistent; its threshold failed. Whether the last-1000-draw window is the
  best future research design is a separate question.
- Evidence against the broad scientific idea: unsupported.
- Possible rescue: a new plan could isolate initialization, folded R-hat scale
  mismatch, transition tuning, and window design without weakening the current
  result or retroactively changing its gate.

## Post-Run Red Team

The strongest alternative explanation is that the fixed `1000`-draw burn-in
window is exposing between-chain scale differences that more burn-in under the
same transition did not resolve; it does not identify whether initialization,
tuning, target geometry, or the window policy is causal. A future result that
passes a predeclared, independently justified convergence design after a
localized repair would overturn rejection of the repaired candidate, but not
this run's recorded failure. The weakest evidence is causal attribution: the
artifact establishes which gate failed, not why it failed.

## Handoff

Phase 7 is terminal at `diagnostic_cap_failure`. Do not retry this academic
campaign and do not execute Phase 8 posterior-recovery/runtime work or NeuTra.
Proceed only to Phase 8 documentation closeout and boundary handoff under
`docs/plans/bayesfilter-hmc-semantic-identity-migration-phase8-closeout-subplan-2026-07-11.md`.

