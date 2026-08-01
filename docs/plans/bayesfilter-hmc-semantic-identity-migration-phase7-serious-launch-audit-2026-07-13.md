# Phase 7 Serious Launch Audit

Date: 2026-07-13

Status: `PASS_FOR_ONE_EXACT_APPROVED_SERIOUS_LAUNCH`

## Authority

The owner supplied exactly:

`I approve AUTHORIZE_PHASE7_TYPED_IDENTITY_TWO_WORKER_CPU_XLA_SERIOUS bound to Phase 7 authority proposal manifest sha256:c1f5709ee64eb898aa74e457553248ef32aac9bdc8a100b3f05f1431eebfa330.`

This authorizes one serious Phase 7 launch only. It grants no Phase 8, NeuTra,
retuning, restart, package, network, default-policy, or scientific-claim
authority.

## Skeptical Audit

| Risk | Verdict |
| --- | --- |
| Wrong baseline | Pass: proposal binds transition `sha256:10d9a9d2...f16d6a` and serious execution `sha256:ceefd154...87e4`, not smoke metrics or a legacy whole-payload hash. |
| Proxy promoted | Pass: smoke R-hat/ESS and no-runtime tests remain explanatory/mechanical only. |
| Missing stop | Pass: identity, evidence, archive/output, authority/claim, finite-value, divergence, XLA, worker, timeout, diagnostic-cap, and artifact stops remain active. |
| Unfair comparison | N/A: no candidate ranking occurs; the fixed transition is screened against predeclared all-parameter gates. |
| Hidden assumptions | Pass: CPU hiding, two persistent workers, four chains, float64, Host XLA/JIT, seeds, thread settings, counts, thresholds, paths, and eight-hour cap are proposal-bound. |
| Stale context | Pass pending immediate verifier: reviewed source/proposal hashes match; all proposal-bound artifacts will be descriptor-pinned again before claim creation. |
| Environment mismatch | Pass pending immediate verifier: launcher fixes CPU hiding and thread variables before import; non-JIT fallback remains forbidden. |
| Artifact cannot answer question | Pass: progress/result, protected samples, log, authority, permanent claim, and terminal output manifest are required. |

Audit verdict: `PASS_FOR_ONE_EXACT_APPROVED_SERIOUS_LAUNCH`.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the fixed typed transition complete serious burn-in and retained sampling under the fixed all-parameter convergence gates? |
| Baseline | Reviewed Phase 5 V2 identities and proposal manifest `sha256:c1f5709e...fa330`. |
| Primary criterion | Every raw parameter passes R-hat `<=1.01`, bulk ESS `>=1000`, and tail ESS `>=400`; all engineering/numerical vetoes and terminal artifacts pass. |
| Promotion veto | Any parameter fails a final retained diagnostic at the cap, or a required diagnostic is missing/nonfinite. |
| Continuation veto | Authority, claim, source/identity, archive/output, finite-value, divergence, XLA/JIT, worker, timeout, public-boundary, or artifact failure. |
| Explanatory only | Intermediate checks, acceptance, timing, PIDs, and descriptive summaries. |
| Not concluded | Posterior recovery, calibrated uncertainty, superiority, production/default/GPU readiness, Phase 8, NeuTra, or broad validity. |

## Pre-Mortem

- The run could appear to pass because diagnostics are computed on the wrong
  chain ordering or source identity. Cheap discriminator: child cache-seal,
  transition/source hashes, worker metadata, and protected sample provenance.
- The run could fail from worker/XLA infrastructure rather than mixing. Cheap
  discriminator: claim stage, worker PIDs, compile traces, infrastructure
  terminal, progress stage, and presence/absence of retained diagnostics.
- The run could fail the current candidate at a diagnostic cap while the
  research direction remains viable. That is candidate rejection, not harness
  invalidity, unless an engineering or numerical continuation veto also fires.
- The run could pass convergence thresholds without posterior recovery. Phase 8
  remains separately unauthorized and is the later recovery question.

## Exact Command

```bash
/home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 \
  scripts/run_hmc_phase7_typed_identity_serious.py \
  --stage burnin_sampling \
  --phase7-serious-authority \
  docs/plans/artifacts/hmc-semantic-identity-migration-2026-07-11/phase7_serious_authority.json
```

No runtime gate, threshold, chain, seed, count, path, or command may be changed
after claim consumption. A terminal failure is recorded; the approval and claim
are never reused.
