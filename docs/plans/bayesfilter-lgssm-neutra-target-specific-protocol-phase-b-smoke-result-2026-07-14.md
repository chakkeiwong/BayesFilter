# LGSSM NeuTra Target-Specific Protocol Phase B Smoke Result

Date: 2026-07-14  
Status: `PASS_WIRING_SMOKES`  
Plan: `docs/plans/bayesfilter-lgssm-neutra-target-specific-training-protocol-amendment-2026-07-14.md`

## Decision

All four predeclared target-specific recipes pass the trusted five-step
GPU/XLA wiring gate. The 500-step nomination screen is technically ready.
These smokes are engineering evidence only and do not nominate a recipe or
support any HMC, posterior, or scientific claim.

## Results

| Recipe | Seed | Artifact hash | Wall time (s) | Target status | Floors | Reload max error | Score max error |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| `source_anchor_lr5e3` | `(20260714,1411)` | `sha256:c6ec1555fd376e6de3ba36613606c799cafb45e5a4c7d9528b5fabe9962bdf96` | 61.27 | valid | 0 | 0 | `3.55e-15` |
| `lower_lr1e3` | `(20260714,1412)` | `sha256:0a280ba9283362c241744310a022c0630c50432f1f9d85026b72cf0169aeb4d5` | 60.99 | valid | 0 | 0 | `1.78e-15` |
| `shallow_2stage_lr5e3` | `(20260714,1413)` | `sha256:e657719ee18f8e91a5d9c92f8f538f20d1169b69c6d2b0800d326d0824143526` | 58.53 | valid | 0 | 0 | `1.78e-15` |
| `wide_2x_lr5e3` | `(20260714,1414)` | `sha256:99036259c6ea6992eba915f33974b8d135c196986565ed9cf07dc96b623c86a7` | 59.65 | valid | 0 | 0 | `1.33e-15` |

Every result reports:

- `passed=true`;
- exact target and adapter identity;
- float64 TensorFlow/XLA output on `/GPU:0` with no soft-placement fallback;
- finite training and target diagnostics;
- zero target-status failures and zero target floors;
- exact trainable/frozen forward and logdet reload;
- explicit frozen transformed score matching the restored trainable reference;
- immutable payload, terminal checkpoint, progress log, and result hashes.

## Commands

Each recipe used the same command shape with a distinct immutable job ID:

```text
MPLCONFIGDIR=/tmp/<recipe> /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py train --job-kind smoke --job-id <recipe>
```

Execution was sequential to avoid GPU contention and ambiguous throughput.

## Artifact Files

| Recipe | Result file SHA-256 |
| --- | --- |
| `source_anchor_lr5e3` | `c6f42d7542404c6a460ae10efa827f847f7842dd654f396ac208163dc4ec8b7e` |
| `lower_lr1e3` | `bd0383c3763a47be25de343ce547a2e1934f96696ff35997f167762036e7a951` |
| `shallow_2stage_lr5e3` | `ab949fed94352df10ba69f1c07a850b930b1bce29eab6b5661b6b01a36a9a21d` |
| `wide_2x_lr5e3` | `1e9f08c93aa7202c4b760ee5696e39323787713a49bc380935c93e1b8aec3ee9` |

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | All four recipe routes compile, train, freeze, reload, and score correctly on trusted GPU/XLA. |
| Numerical/sampler validity | Target and transformed-score mechanics pass at smoke scale; no tuning or HMC result exists. |
| Scientific interpretation | No recipe is nominated and no learned-transport claim is supported by five steps. |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Allow the reviewed 500-step screen after compute approval. | Four wiring smokes passed. | No target, finite-state, device, XLA, reload, or score veto. | Five steps say nothing about recipe quality or long-budget optimization. | Run all four 500-step screen arms and common held-out evaluation under the frozen contract. | No recipe nomination, HMC readiness, posterior correctness, superiority, robustness, or default readiness. |

## Post-Run Red Team

Strongest alternative explanation: the smoke passes because five steps are too
short to expose chronic clipping, target-region drift, or capacity-specific
optimization failure. The 500-step screen is designed to expose those failures.

What would overturn the handoff: any screen arm may still fail individually;
a common target/math failure or zero surviving arms stops the campaign.

Weakest evidence: no held-out recipe comparison or downstream HMC has run.
